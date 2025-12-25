--[[
    Semantic Search - Configuration

    Configuration for the Semantic Search plugin.
]]

local LrPathUtils = import 'LrPathUtils'
local LrFileUtils = import 'LrFileUtils'

local Config = {}

-- Server configuration
Config.SERVER_HOST = "127.0.0.1"
Config.SERVER_PORT = 5555
Config.SERVER_URL = "http://" .. Config.SERVER_HOST .. ":" .. Config.SERVER_PORT

-- Default search parameters
Config.DEFAULT_MAX_RESULTS = 500
Config.DEFAULT_MIN_SCORE = 0.20

-- Collection naming
Config.COLLECTION_SET_NAME = "0_Semantic_Searches"
Config.MAX_COLLECTION_NAME_LENGTH = 50

-- Get the temp directory for results files
-- Uses catalog directory + /tmp/
function Config.getResultsDir(catalog)
    local catalogPath = catalog:getPath()
    local catalogDir = LrPathUtils.parent(catalogPath)
    local tmpDir = LrPathUtils.child(catalogDir, "tmp")

    -- Create if doesn't exist
    if not LrFileUtils.exists(tmpDir) then
        LrFileUtils.createDirectory(tmpDir)
    end

    return tmpDir
end

-- Generate a results file path with timestamp
function Config.getResultsFilePath(catalog)
    local tmpDir = Config.getResultsDir(catalog)
    local timestamp = os.date("%Y%m%d_%H%M%S")
    return LrPathUtils.child(tmpDir, "semantic_search_" .. timestamp .. ".json")
end

-- Truncate query for collection name
function Config.makeCollectionName(query)
    local name = query
    if #name > Config.MAX_COLLECTION_NAME_LENGTH then
        name = name:sub(1, Config.MAX_COLLECTION_NAME_LENGTH - 3) .. "..."
    end
    return name
end

return Config
