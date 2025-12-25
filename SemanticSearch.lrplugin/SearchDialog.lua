--[[
    Semantic Search - Search Dialog

    Main entry point for the semantic search plugin.
    Shows a dialog to enter search parameters, calls the search server,
    and creates a collection with the results.
]]

local LrApplication = import 'LrApplication'
local LrBinding = import 'LrBinding'
local LrColor = import 'LrColor'
local LrDialogs = import 'LrDialogs'
local LrFunctionContext = import 'LrFunctionContext'
local LrHttp = import 'LrHttp'
local LrPathUtils = import 'LrPathUtils'
local LrTasks = import 'LrTasks'
local LrView = import 'LrView'
local LrLogger = import 'LrLogger'

local Config = require 'Config'

local logger = LrLogger('SemanticSearch')
logger:enable('logfile')

-- JSON encoding (simple implementation for our needs)
local function jsonEncode(tbl)
    local function encodeValue(val)
        local t = type(val)
        if t == "string" then
            -- Escape special characters
            local escaped = val:gsub('\\', '\\\\')
                              :gsub('"', '\\"')
                              :gsub('\n', '\\n')
                              :gsub('\r', '\\r')
                              :gsub('\t', '\\t')
            return '"' .. escaped .. '"'
        elseif t == "number" then
            return tostring(val)
        elseif t == "boolean" then
            return val and "true" or "false"
        elseif t == "table" then
            -- Check if array or object
            local isArray = #val > 0 or next(val) == nil
            if isArray then
                local parts = {}
                for _, v in ipairs(val) do
                    table.insert(parts, encodeValue(v))
                end
                return "[" .. table.concat(parts, ",") .. "]"
            else
                local parts = {}
                for k, v in pairs(val) do
                    table.insert(parts, '"' .. tostring(k) .. '":' .. encodeValue(v))
                end
                return "{" .. table.concat(parts, ",") .. "}"
            end
        elseif t == "nil" then
            return "null"
        else
            return "null"
        end
    end
    return encodeValue(tbl)
end

-- JSON decoding (simple implementation)
local function jsonDecode(str)
    -- Use Lua pattern matching for simple JSON
    -- This handles our expected response format
    local results = {}

    -- Extract count
    local count = str:match('"count"%s*:%s*(%d+)')
    results.count = tonumber(count) or 0

    -- Extract elapsed_ms
    local elapsed = str:match('"elapsed_ms"%s*:%s*([%d%.]+)')
    results.elapsed_ms = tonumber(elapsed) or 0

    -- Extract error if present
    local error = str:match('"error"%s*:%s*"([^"]*)"')
    results.error = error

    -- Extract results array - get paths and scores
    results.matches = {}
    for path, score in str:gmatch('"path"%s*:%s*"([^"]*)"%s*,%s*"score"%s*:%s*([%d%.]+)') do
        table.insert(results.matches, { path = path, score = tonumber(score) })
    end
    -- Also try reverse order (score before path)
    for score, path in str:gmatch('"score"%s*:%s*([%d%.]+)%s*,%s*"path"%s*:%s*"([^"]*)"') do
        table.insert(results.matches, { path = path, score = tonumber(score) })
    end

    return results
end

-- Get selected photo paths
local function getSelectedPhotoPaths()
    local catalog = LrApplication.activeCatalog()
    local selectedPhotos = catalog:getTargetPhotos()
    local paths = {}

    for _, photo in ipairs(selectedPhotos) do
        local path = photo:getRawMetadata('path')
        if path then
            table.insert(paths, path)
        end
    end

    return paths
end

-- Get all photos in catalog
local function getAllPhotoPaths()
    local catalog = LrApplication.activeCatalog()
    local allPhotos = catalog:getAllPhotos()
    local paths = {}

    for _, photo in ipairs(allPhotos) do
        local path = photo:getRawMetadata('path')
        if path then
            table.insert(paths, path)
        end
    end

    return paths
end

-- Call the search server
local function callSearchServer(query, limit, minScore, returnAll, constrainPaths, resultsFile)
    local url = Config.SERVER_URL .. "/search"

    local requestBody = {
        query = query,
        limit = limit,
        min_score = minScore,
        return_all = returnAll,
        results_file = resultsFile
    }

    if constrainPaths and #constrainPaths > 0 then
        requestBody.paths = constrainPaths
    end

    local jsonBody = jsonEncode(requestBody)
    logger:info("Search request: " .. jsonBody:sub(1, 500))

    local headers = {
        { field = "Content-Type", value = "application/json" }
    }

    local response, respHeaders = LrHttp.post(url, jsonBody, headers, "POST", 120)

    if not response then
        return nil, "Failed to connect to search server. Is it running?"
    end

    logger:info("Search response: " .. response:sub(1, 500))
    return jsonDecode(response), nil
end

-- Get or create the collection set for search results
local function getOrCreateCollectionSet(catalog)
    local setName = Config.COLLECTION_SET_NAME
    local collectionSet = nil

    -- Look for existing set
    for _, set in ipairs(catalog:getChildCollectionSets()) do
        if set:getName() == setName then
            collectionSet = set
            break
        end
    end

    -- Create if not found (in separate write transaction)
    if not collectionSet then
        catalog:withWriteAccessDo("Create collection set", function()
            collectionSet = catalog:createCollectionSet(setName, nil, true)
        end)
    end

    return collectionSet
end

-- Create a collection with the search results
local function createSearchCollection(query, matches)
    local catalog = LrApplication.activeCatalog()
    local collectionName = Config.makeCollectionName(query)

    -- Find photos by path and add to collection
    local photosToAdd = {}

    catalog:withReadAccessDo(function()
        for _, match in ipairs(matches) do
            local photo = catalog:findPhotoByPath(match.path)
            if photo then
                table.insert(photosToAdd, photo)
            end
        end
    end)

    if #photosToAdd == 0 then
        return nil, "No matching photos found in catalog"
    end

    -- Get or create the collection set first (may do its own write transaction)
    local collectionSet = getOrCreateCollectionSet(catalog)

    -- Now create/update collection in a separate write transaction
    local collection = nil
    catalog:withWriteAccessDo("Create search collection", function()
        -- Check if collection already exists in the set
        for _, coll in ipairs(collectionSet:getChildCollections()) do
            if coll:getName() == collectionName then
                collection = coll
                -- Clear existing photos
                local existingPhotos = collection:getPhotos()
                if #existingPhotos > 0 then
                    collection:removePhotos(existingPhotos)
                end
                break
            end
        end

        -- Create new if not found
        if not collection then
            collection = catalog:createCollection(collectionName, collectionSet, true)
        end

        if collection then
            collection:addPhotos(photosToAdd)
        end
    end)

    return collection, nil
end

-- Main function
LrTasks.startAsyncTask(function()
    LrFunctionContext.callWithContext('searchDialog', function(context)
        local catalog = LrApplication.activeCatalog()
        local props = LrBinding.makePropertyTable(context)

        -- Initialize properties
        props.query = ""
        props.maxResults = Config.DEFAULT_MAX_RESULTS
        props.minScore = Config.DEFAULT_MIN_SCORE
        props.returnAll = false

        -- Check selection (treat 1 photo as "no selection" since it's likely accidental)
        local selectedPaths = getSelectedPhotoPaths()
        local hasSelection = #selectedPaths > 1
        props.searchAll = not hasSelection
        props.selectionInfo = hasSelection
            and string.format("Searching %d selected photos", #selectedPaths)
            or "Searching All Photographs"

        local f = LrView.osFactory()

        local contents = f:column {
            spacing = f:control_spacing(),
            bind_to_object = props,

            f:row {
                f:static_text {
                    title = "Search query:",
                    alignment = 'right',
                    width = 100,
                },
                f:edit_field {
                    value = LrView.bind('query'),
                    width_in_chars = 40,
                    immediate = true,
                },
            },

            f:row {
                f:static_text {
                    title = "",
                    width = 100,
                },
                f:static_text {
                    title = LrView.bind('selectionInfo'),
                    text_color = LrColor(0.3, 0.3, 0.3),
                },
                f:checkbox {
                    title = "Search all photos",
                    value = LrView.bind('searchAll'),
                },
            },

            f:separator { fill_horizontal = 1 },

            f:row {
                f:static_text {
                    title = "Max results:",
                    alignment = 'right',
                    width = 100,
                },
                f:edit_field {
                    value = LrView.bind('maxResults'),
                    width_in_digits = 6,
                    min = 1,
                    max = 100000,
                    increment = 100,
                    precision = 0,
                },
                f:checkbox {
                    title = "Return all matches",
                    value = LrView.bind('returnAll'),
                },
            },

            f:row {
                f:static_text {
                    title = "Min similarity:",
                    alignment = 'right',
                    width = 100,
                },
                f:edit_field {
                    value = LrView.bind('minScore'),
                    width_in_digits = 5,
                    min = 0.0,
                    max = 1.0,
                    increment = 0.05,
                    precision = 2,
                },
                f:static_text {
                    title = "(0.0 - 1.0)",
                    text_color = LrColor(0.5, 0.5, 0.5),
                },
            },
        }

        local result = LrDialogs.presentModalDialog({
            title = 'Semantic Search',
            contents = contents,
            actionVerb = 'Search',
        })

        if result ~= 'ok' then
            return
        end

        -- Validate query
        local query = props.query
        if not query or query:match("^%s*$") then
            LrDialogs.message("Semantic Search", "Please enter a search query.", "warning")
            return
        end

        -- Determine paths to search
        local constrainPaths = nil
        if hasSelection and not props.searchAll then
            constrainPaths = selectedPaths
        end

        -- Get results file path
        local resultsFile = Config.getResultsFilePath(catalog)

        -- Show progress dialog
        LrDialogs.showBezel("Searching...")

        -- Call search server
        local response, err = callSearchServer(
            query,
            props.maxResults,
            props.minScore,
            props.returnAll,
            constrainPaths,
            resultsFile
        )

        if err then
            LrDialogs.message("Semantic Search", err, "critical")
            return
        end

        if response.error then
            LrDialogs.message("Semantic Search", "Server error: " .. response.error, "critical")
            return
        end

        if response.count == 0 then
            LrDialogs.message("Semantic Search", "No matching images found.", "info")
            return
        end

        -- Create collection with results
        local collection, collErr = createSearchCollection(query, response.matches)

        if collErr then
            LrDialogs.message("Semantic Search", collErr, "warning")
            return
        end

        -- Show success message
        local message = string.format(
            "Found %d matching images in %.0fms\n\nCollection created: %s",
            response.count,
            response.elapsed_ms,
            Config.makeCollectionName(query)
        )

        LrDialogs.message("Semantic Search", message, "info")

        -- Select the new collection in Library
        if collection then
            catalog:withWriteAccessDo("Select collection", function()
                catalog:setActiveSources({ collection })
            end)
        end
    end)
end)
